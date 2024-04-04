import cx from 'clsx'
import { useState } from 'react'
import { Table, ScrollArea, ActionIcon, Text, Group } from '@mantine/core'
import classes from './FileList.module.css'
import { IconExternalLink, IconX } from '@tabler/icons-react'
import { FilePreviewModal } from './FilePreviewModal.tsx'
import { FileWithId } from '../App.tsx'

export interface FileListProps {
  filesWithId: FileWithId[]
  setFilesWithId: any
}

export const FileList = ({ filesWithId, setFilesWithId }: FileListProps) => {
  const [scrolled, setScrolled] = useState(false)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [previewModalOpen, setPreviewModalOpen] = useState(false)

  const handleRemoveFile = (id: string) => {
    setFilesWithId((prevFiles: FileWithId[]) =>
      prevFiles.filter((fileWithId: FileWithId) => fileWithId.id !== id)
    )
  }

  const handleFileClick = (file: File) => {
    setSelectedFile(file)
    setPreviewModalOpen(true)
  }

  // @ts-ignore
  const filesExpand = filesWithId.map((fileWithId: FileWithId) => (
    <Table.Tr key={fileWithId.file.name}>
      <Table.Td className={classes.tdName}>
        <Group>
          <Text>{fileWithId.file.name}</Text>
          <ActionIcon
            onClick={() => handleFileClick(fileWithId.file)}
            variant="light"
          >
            <IconExternalLink />
          </ActionIcon>
        </Group>
      </Table.Td>
      <Table.Td>
        <ActionIcon
          color="red"
          variant="filled"
          onClick={() => handleRemoveFile(fileWithId.id)}
        >
          <IconX />
        </ActionIcon>
      </Table.Td>
    </Table.Tr>
  ))

  return (
    <>
      <ScrollArea
        onScrollPositionChange={({ y }) => setScrolled(y !== 0)}
        className={cx(classes.table)}
      >
        <Table>
          <Table.Thead
            className={`${cx(classes.header, { [classes.scrolled]: scrolled })}`}
          >
            <Table.Tr>
              <Table.Th className={classes.thName}>Name</Table.Th>
              <Table.Th className={classes.thRemove}>Remove</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>{filesExpand}</Table.Tbody>
        </Table>
      </ScrollArea>
      {selectedFile && (
        <FilePreviewModal
          file={selectedFile}
          isModalOpen={previewModalOpen}
          setIsModalOpen={setPreviewModalOpen}
        />
      )}
    </>
  )
}
