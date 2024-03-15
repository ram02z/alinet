import cx from 'clsx'
import { useState } from 'react'
import { Table, ScrollArea } from '@mantine/core'
import classes from './FileList.module.css'

export interface FileListProps {
  files: File[]
  setFiles: any
}

export const FileList = ({ files, setFiles }: FileListProps) => {
  const [scrolled, setScrolled] = useState(false)

  const handleRemoveFile = (index: number) => {
    setFiles((prevFiles: File[]) => prevFiles.filter((_, i) => i !== index))
    console.log(files)
  }

  const filesExpand = files.map((file, index) => (
    <Table.Tr key={file.name}>
      <Table.Td>{file.name}</Table.Td>
      <Table.Td>
        <button onClick={() => handleRemoveFile(index)} />
      </Table.Td>
    </Table.Tr>
  ))

  return (
    <ScrollArea
      h={300}
      w={800}
      onScrollPositionChange={({ y }) => setScrolled(y !== 0)}
    >
      <Table>
        <Table.Thead
          className={cx(classes.header, { [classes.scrolled]: scrolled })}
        >
          <Table.Tr>
            <Table.Th>Name</Table.Th>
            <Table.Th>Remove</Table.Th>
          </Table.Tr>
        </Table.Thead>
        <Table.Tbody>{filesExpand}</Table.Tbody>
      </Table>
    </ScrollArea>
  )
}
