import cx from 'clsx'
import {
  Table,
  Checkbox,
  ScrollArea,
  rem,
  Group,
  Center,
  Tooltip,
} from '@mantine/core'
import classes from './QuestionTable.module.css'
import { Question } from '../App'
import { useState } from 'react'
import {
  IconSelector,
  IconChevronDown,
  IconChevronUp,
  IconZoomQuestion,
} from '@tabler/icons-react'
export interface QuestionTableProps {
  selection: string[]
  setSelection: any
  questions: Question[]
}
import { Slider } from '@mantine/core'

export const QuestionTable = ({
  selection,
  setSelection,
  questions,
}: QuestionTableProps) => {
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc' | null>(
    null
  )

  const [similarityThreshold, setSimilarityThreshold] = useState(0)

  const Icon =
    sortDirection != null
      ? sortDirection === 'desc'
        ? IconChevronDown
        : IconChevronUp
      : IconSelector

  const sortedQuestions = [...questions].sort((a, b) => {
    if (sortDirection === 'asc') {
      return a.score - b.score
    }
    if (sortDirection === 'desc') {
      return b.score - a.score
    }
    return 0
  })

  const toggleSortDirection = () => {
    setSortDirection((currentSortDirection) => {
      if (currentSortDirection === 'asc') {
        return 'desc'
      } else {
        return 'asc'
      }
    })
  }

  const toggleRow = (id: string) => {
    setSelection((current: string[]) =>
      current.includes(id)
        ? current.filter((item: string) => item !== id)
        : [...current, id]
    )
  }

  const toggleAll = () => {
    setSelection((current: string[]) =>
      current.length === questions.length
        ? []
        : questions.map((item) => item.id)
    )
  }

  const rows: any = sortedQuestions.map((item: Question) => {
    if (item.score >= similarityThreshold) {
      const selected = selection.includes(item.id)
      return (
        <Table.Tr
          key={item.id}
          className={cx({ [classes.rowSelected]: selected })}
        >
          <Table.Td>
            <Checkbox
              checked={selection.includes(item.id)}
              onChange={() => toggleRow(item.id)}
            />
          </Table.Td>
          <Table.Td>{item.text}</Table.Td>
          <Table.Td>{item.score.toFixed(2)}</Table.Td>
        </Table.Tr>
      )
    }
  })

  return (
    <>
      <div className={classes.config_container}>
        <div id={classes.title}>Similarity Threshold</div>
        <Slider
          value={similarityThreshold}
          onChange={setSimilarityThreshold}
          color="blue"
          size="xl"
          min={0}
          max={1}
          marks={[
            { value: 0.25, label: '0.25' },
            { value: 0.5, label: '0.50' },
            { value: 0.75, label: '0.75' },
          ]}
          step={0.01}
          className={classes.similarity_slider}
        />
      </div>

      <ScrollArea>
        <Table verticalSpacing="md">
          <Table.Thead>
            <Table.Tr>
              <Table.Th style={{ width: rem(40) }}>
                <Checkbox
                  onChange={toggleAll}
                  checked={
                    questions.length > 0 &&
                    selection.length === questions.length
                  }
                  indeterminate={
                    selection.length > 0 &&
                    selection.length !== questions.length
                  }
                />
              </Table.Th>
              <Table.Th>Questions</Table.Th>
              <Table.Th>
                <Group gap="0">
                  <Tooltip label="The score indicates the degree to which the context chunk, from which the question was derived, aligns with the learning material's content.">
                    <IconZoomQuestion size={'1.2rem'} />
                  </Tooltip>
                  Similarity Score
                  <Center>
                    <Icon
                      onClick={toggleSortDirection}
                      style={{
                        width: rem(20),
                        height: rem(20),
                        cursor: 'pointer',
                        marginLeft: rem(5),
                      }}
                      stroke={1.5}
                    />
                  </Center>
                </Group>
              </Table.Th>
            </Table.Tr>
          </Table.Thead>

          <Table.Tbody>{rows}</Table.Tbody>
        </Table>
      </ScrollArea>
    </>
  )
}
